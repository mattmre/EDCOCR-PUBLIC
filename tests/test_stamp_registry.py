"""Unit tests for stamp registry and contracts.

Tests the stamp operation registry, configuration validation, and result models.
Does not require PyMuPDF - uses mock stamp implementations.

Run with: python -m pytest tests/test_stamp_registry.py -v
"""

import os
from typing import Any

from ocr_distributed.stamps import (
    StampConfig,
    StampError,
    StampOperation,
    StampPlacement,
    StampRegistry,
    StampResult,
    StampValidationError,
    get_stamp_registry,
)

# --- Mock Stamp Implementations ---


class MockBatesStamp(StampOperation):
    """Mock stamp for testing - Bates numbering."""

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "bates",
            "description": "Apply Bates numbering to documents",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "parameters": {
                "prefix": {
                    "type": "str",
                    "description": "Bates prefix (e.g., 'PROD')",
                    "required": False,
                },
                "start": {
                    "type": "int",
                    "description": "Starting Bates number",
                    "required": True,
                },
                "width": {
                    "type": "int",
                    "description": "Zero-padding width",
                    "required": False,
                },
            },
        }

    def validate_config(self, config: StampConfig) -> list[str]:
        errors = []
        start = config.params.get("start")
        if start is None:
            errors.append("start parameter is required")
        elif not isinstance(start, int) or start < 1:
            errors.append(f"start must be positive integer, got {start}")

        width = config.params.get("width", 6)
        if not isinstance(width, int) or width < 1 or width > 12:
            errors.append(f"width must be 1-12, got {width}")

        return errors

    def execute(
        self, input_path: str, output_path: str, config: StampConfig
    ) -> StampResult:
        if not os.path.exists(input_path):
            raise StampValidationError(f"Input file not found: {input_path}")

        start = config.params.get("start", 1)
        prefix = config.params.get("prefix", "")
        width = config.params.get("width", 6)

        # Mock Bates generation for 10 pages
        stamp_values = [
            f"{prefix}{str(start + i).zfill(width)}" for i in range(10)
        ]

        return StampResult(
            success=True,
            output_path=output_path,
            pages_stamped=10,
            stamp_values=stamp_values,
            metadata={"prefix": prefix, "start": start, "width": width},
        )


class MockConfidentialityStamp(StampOperation):
    """Mock stamp for testing - confidentiality designations."""

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "confidentiality",
            "description": "Apply confidentiality designation",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "parameters": {
                "designation": {
                    "type": "str",
                    "description": "Designation text",
                    "required": True,
                },
            },
        }

    def validate_config(self, config: StampConfig) -> list[str]:
        errors = []
        designation = config.params.get("designation")
        if not designation:
            errors.append("designation parameter is required")
        elif designation not in ["CONFIDENTIAL", "HIGHLY CONFIDENTIAL", "ATTORNEYS' EYES ONLY"]:
            errors.append(f"Invalid designation: {designation}")
        return errors

    def execute(
        self, input_path: str, output_path: str, config: StampConfig
    ) -> StampResult:
        designation = config.params.get("designation")
        return StampResult(
            success=True,
            output_path=output_path,
            pages_stamped=10,
            stamp_values=[designation] * 10,
            metadata={"designation": designation},
        )


# --- StampPlacement Tests ---


class TestStampPlacement:
    """Tests for StampPlacement enum."""

    def test_all_placements_defined(self):
        expected = [
            "TOP_LEFT", "TOP_CENTER", "TOP_RIGHT",
            "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT",
            "CENTER",
        ]
        for name in expected:
            assert hasattr(StampPlacement, name)

    def test_placement_values(self):
        assert StampPlacement.TOP_LEFT.value == "top_left"
        assert StampPlacement.BOTTOM_RIGHT.value == "bottom_right"
        assert StampPlacement.CENTER.value == "center"

    def test_placement_is_string_enum(self):
        assert isinstance(StampPlacement.TOP_LEFT, str)
        assert StampPlacement.TOP_LEFT == "top_left"


# --- StampConfig Tests ---


class TestStampConfig:
    """Tests for StampConfig dataclass."""

    def test_minimal_config(self):
        config = StampConfig(operation_name="bates")
        assert config.operation_name == "bates"
        assert config.placement == StampPlacement.BOTTOM_RIGHT
        assert config.params == {}
        assert config.validate_input is True
        assert config.check_overlap is True

    def test_full_config(self):
        config = StampConfig(
            operation_name="bates",
            placement=StampPlacement.TOP_RIGHT,
            params={"prefix": "PROD", "start": 1},
            validate_input=False,
            check_overlap=False,
        )
        assert config.operation_name == "bates"
        assert config.placement == StampPlacement.TOP_RIGHT
        assert config.params["prefix"] == "PROD"
        assert config.validate_input is False
        assert config.check_overlap is False

    def test_empty_operation_name_raises(self):
        try:
            StampConfig(operation_name="")
            assert False, "Should raise StampValidationError"
        except StampValidationError as e:
            assert "operation_name is required" in str(e)

    def test_invalid_placement_type_raises(self):
        try:
            StampConfig(operation_name="bates", placement="top_left")  # type: ignore
            assert False, "Should raise StampValidationError"
        except StampValidationError as e:
            assert "must be StampPlacement enum" in str(e)


# --- StampResult Tests ---


class TestStampResult:
    """Tests for StampResult dataclass."""

    def test_success_result(self):
        result = StampResult(
            success=True,
            output_path="/tmp/output.pdf",
            pages_stamped=5,
            stamp_values=["PROD000001", "PROD000002"],
        )
        assert result.success is True
        assert result.output_path == "/tmp/output.pdf"
        assert result.pages_stamped == 5
        assert len(result.stamp_values) == 2
        assert result.error_message is None

    def test_failure_result(self):
        result = StampResult(
            success=False,
            error_message="Invalid Bates configuration",
        )
        assert result.success is False
        assert result.error_message == "Invalid Bates configuration"
        assert result.output_path is None

    def test_success_without_output_path_raises(self):
        try:
            StampResult(success=True, pages_stamped=5)
            assert False, "Should raise StampValidationError"
        except StampValidationError as e:
            assert "output_path is required" in str(e)

    def test_failure_without_error_message_raises(self):
        try:
            StampResult(success=False)
            assert False, "Should raise StampValidationError"
        except StampValidationError as e:
            assert "error_message is required" in str(e)

    def test_result_with_warnings(self):
        result = StampResult(
            success=True,
            output_path="/tmp/output.pdf",
            pages_stamped=10,
            stamp_values=["PROD000001"],
            warnings=["Overlap detected on page 3", "Bates near edge on page 7"],
        )
        assert len(result.warnings) == 2
        assert "Overlap detected" in result.warnings[0]


# --- StampOperation Tests ---


class TestStampOperation:
    """Tests for StampOperation abstract base class."""

    def test_mock_operation_metadata(self):
        op = MockBatesStamp()
        metadata = op.get_metadata()
        assert metadata["name"] == "bates"
        assert metadata["version"] == "1.0.0"
        assert ".pdf" in metadata["supported_formats"]

    def test_mock_operation_validate_config_success(self):
        op = MockBatesStamp()
        config = StampConfig(
            operation_name="bates",
            params={"start": 1, "width": 6},
        )
        errors = op.validate_config(config)
        assert errors == []

    def test_mock_operation_validate_config_failure(self):
        op = MockBatesStamp()
        config = StampConfig(operation_name="bates", params={"start": -1})
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "must be positive integer" in errors[0]

    def test_supports_format(self):
        op = MockBatesStamp()
        assert op.supports_format(".pdf") is True
        assert op.supports_format(".PDF") is True
        assert op.supports_format(".png") is False

    def test_confidentiality_stamp_validate(self):
        op = MockConfidentialityStamp()
        config = StampConfig(
            operation_name="confidentiality",
            params={"designation": "CONFIDENTIAL"},
        )
        errors = op.validate_config(config)
        assert errors == []

    def test_confidentiality_stamp_invalid_designation(self):
        op = MockConfidentialityStamp()
        config = StampConfig(
            operation_name="confidentiality",
            params={"designation": "SECRET"},
        )
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "Invalid designation" in errors[0]


# --- StampRegistry Tests ---


class TestStampRegistry:
    """Tests for StampRegistry."""

    def setup_method(self):
        """Create fresh registry for each test."""
        self.registry = StampRegistry()

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
        op = MockBatesStamp()
        self.registry.register(op)
        assert self.registry.list_operations() == ["bates"]

    def test_register_multiple_operations(self):
        self.registry.register(MockBatesStamp())
        self.registry.register(MockConfidentialityStamp())
        ops = self.registry.list_operations()
        assert len(ops) == 2
        assert "bates" in ops
        assert "confidentiality" in ops

    def test_register_duplicate_name_raises(self):
        self.registry.register(MockBatesStamp())
        try:
            self.registry.register(MockBatesStamp())
            assert False, "Should raise StampValidationError"
        except StampValidationError as e:
            assert "already registered" in str(e)

    def test_get_operation(self):
        op = MockBatesStamp()
        self.registry.register(op)
        retrieved = self.registry.get("bates")
        assert retrieved is op

    def test_get_nonexistent_operation(self):
        result = self.registry.get("nonexistent")
        assert result is None

    def test_get_metadata(self):
        self.registry.register(MockBatesStamp())
        metadata = self.registry.get_metadata("bates")
        assert metadata is not None
        assert metadata["name"] == "bates"
        assert metadata["version"] == "1.0.0"

    def test_get_metadata_nonexistent(self):
        result = self.registry.get_metadata("nonexistent")
        assert result is None

    def test_list_all_metadata(self):
        self.registry.register(MockBatesStamp())
        self.registry.register(MockConfidentialityStamp())
        all_metadata = self.registry.list_all_metadata()
        assert len(all_metadata) == 2
        names = [m["name"] for m in all_metadata]
        assert "bates" in names
        assert "confidentiality" in names

    def test_list_all_metadata_sorted(self):
        # Register in reverse order
        self.registry.register(MockConfidentialityStamp())
        self.registry.register(MockBatesStamp())
        all_metadata = self.registry.list_all_metadata()
        names = [m["name"] for m in all_metadata]
        # Should be sorted alphabetically
        assert names == sorted(names)

    def test_list_all_metadata_uses_snapshot_after_lock_release(self):
        self.registry.register(MockBatesStamp())
        self.registry.register(MockConfidentialityStamp())
        self.registry._operations = self.ClearAfterItemsDict(self.registry._operations)

        all_metadata = self.registry.list_all_metadata()

        assert [m["name"] for m in all_metadata] == ["bates", "confidentiality"]
        assert self.registry._operations == {}

    def test_clear_registry(self):
        self.registry.register(MockBatesStamp())
        assert len(self.registry.list_operations()) == 1
        self.registry.clear()
        assert len(self.registry.list_operations()) == 0

    def test_register_invalid_type_raises(self):
        try:
            self.registry.register("not an operation")  # type: ignore
            assert False, "Should raise StampValidationError"
        except StampValidationError as e:
            assert "must inherit from StampOperation" in str(e)

    def test_register_operation_without_name_raises(self):
        class BadOperation(StampOperation):
            def get_metadata(self):
                return {"description": "Missing name"}

            def validate_config(self, config):
                return []

            def execute(self, input_path, output_path, config):
                return StampResult(success=False, error_message="Not implemented")

        try:
            self.registry.register(BadOperation())
            assert False, "Should raise StampValidationError"
        except StampValidationError as e:
            assert "must include 'name'" in str(e)

    def test_register_operation_with_empty_name_raises(self):
        class BadOperation(StampOperation):
            def get_metadata(self):
                return {"name": "   "}

            def validate_config(self, config):
                return []

            def execute(self, input_path, output_path, config):
                return StampResult(success=False, error_message="Not implemented")

        try:
            self.registry.register(BadOperation())
            assert False, "Should raise StampValidationError"
        except StampValidationError as e:
            assert "non-empty string" in str(e)


# --- Global Registry Tests ---


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    def test_get_stamp_registry_returns_singleton(self):
        registry1 = get_stamp_registry()
        registry2 = get_stamp_registry()
        assert registry1 is registry2

    def test_global_registry_persists_across_calls(self):
        registry = get_stamp_registry()
        registry.clear()  # Start clean
        
        registry.register(MockBatesStamp())
        
        # Get registry again - should have same operation
        registry2 = get_stamp_registry()
        assert "bates" in registry2.list_operations()
        
        # Clean up
        registry.clear()


# --- Exception Tests ---


class TestStampExceptions:
    """Tests for stamp exception hierarchy."""

    def test_stamp_error_inheritance(self):
        err = StampError("test error")
        assert isinstance(err, Exception)

    def test_stamp_validation_error_inheritance(self):
        err = StampValidationError("validation failed")
        assert isinstance(err, StampError)
        assert isinstance(err, Exception)
