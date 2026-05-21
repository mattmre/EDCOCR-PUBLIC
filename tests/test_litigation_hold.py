"""Tests for centralized LITIGATION_HOLD enforcement (C-10).

Validates:
- ``is_litigation_hold_active()`` recognises all truthy/falsy env var values
- ``check_litigation_hold()`` writes to stderr and returns True when active
- All deletion management commands import from the shared module
- All Celery cleanup tasks import from the shared module
"""

import importlib
import os
import unittest
from unittest.mock import patch


class TestIsLitigationHoldActive(unittest.TestCase):
    """Test the ``is_litigation_hold_active()`` function."""

    def _get_func(self):
        """Import the function under test (deferred to avoid Django setup)."""
        # litigation_hold.py is pure-Python, no Django ORM imports
        mod_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "coordinator",
            "jobs",
            "litigation_hold.py",
        )
        mod_path = os.path.normpath(mod_path)
        spec = importlib.util.spec_from_file_location(
            "jobs.litigation_hold", mod_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.is_litigation_hold_active, mod.check_litigation_hold

    # -- Truthy values -------------------------------------------------------

    def test_truthy_1(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "1"}):
            self.assertTrue(is_active())

    def test_truthy_true_lower(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "true"}):
            self.assertTrue(is_active())

    def test_truthy_true_upper(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "TRUE"}):
            self.assertTrue(is_active())

    def test_truthy_true_mixed(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "True"}):
            self.assertTrue(is_active())

    def test_truthy_yes_lower(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "yes"}):
            self.assertTrue(is_active())

    def test_truthy_yes_upper(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "YES"}):
            self.assertTrue(is_active())

    def test_truthy_yes_mixed(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "Yes"}):
            self.assertTrue(is_active())

    # -- Falsy values --------------------------------------------------------

    def test_falsy_unset(self):
        is_active, _ = self._get_func()
        env = os.environ.copy()
        env.pop("LITIGATION_HOLD", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(is_active())

    def test_falsy_empty(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": ""}):
            self.assertFalse(is_active())

    def test_falsy_false(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "false"}):
            self.assertFalse(is_active())

    def test_falsy_0(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "0"}):
            self.assertFalse(is_active())

    def test_falsy_no(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "no"}):
            self.assertFalse(is_active())

    def test_falsy_random_string(self):
        is_active, _ = self._get_func()
        with patch.dict(os.environ, {"LITIGATION_HOLD": "nope"}):
            self.assertFalse(is_active())


class TestCheckLitigationHold(unittest.TestCase):
    """Test the ``check_litigation_hold()`` command helper."""

    def _get_func(self):
        mod_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "coordinator",
            "jobs",
            "litigation_hold.py",
        )
        mod_path = os.path.normpath(mod_path)
        spec = importlib.util.spec_from_file_location(
            "jobs.litigation_hold", mod_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.check_litigation_hold

    def test_returns_true_when_active(self):
        check = self._get_func()

        class FakeCommand:
            stderr = type("FakeStderr", (), {"write": staticmethod(lambda msg: None)})()

        with patch.dict(os.environ, {"LITIGATION_HOLD": "true"}):
            self.assertTrue(check(FakeCommand()))

    def test_returns_false_when_inactive(self):
        check = self._get_func()

        class FakeCommand:
            stderr = type("FakeStderr", (), {"write": staticmethod(lambda msg: None)})()

        env = os.environ.copy()
        env.pop("LITIGATION_HOLD", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(check(FakeCommand()))

    def test_writes_to_stderr_when_active(self):
        check = self._get_func()
        captured = []

        class FakeCommand:
            class stderr:
                @staticmethod
                def write(msg):
                    captured.append(msg)

        with patch.dict(os.environ, {"LITIGATION_HOLD": "1"}):
            check(FakeCommand())

        self.assertEqual(len(captured), 1)
        self.assertIn("LITIGATION_HOLD", captured[0])
        self.assertIn("blocked", captured[0])

    def test_no_stderr_when_inactive(self):
        check = self._get_func()
        captured = []

        class FakeCommand:
            class stderr:
                @staticmethod
                def write(msg):
                    captured.append(msg)

        env = os.environ.copy()
        env.pop("LITIGATION_HOLD", None)
        with patch.dict(os.environ, env, clear=True):
            check(FakeCommand())

        self.assertEqual(len(captured), 0)


class TestAllDeletionCommandsUseSharedModule(unittest.TestCase):
    """Verify that all deletion management commands import from the shared module.

    This test reads the source files directly to confirm the DRY refactor.
    """

    _COMMANDS_DIR = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "coordinator",
            "jobs",
            "management",
            "commands",
        )
    )

    _DELETION_COMMANDS = [
        "cleanup_old_jobs.py",
        "cleanup_output.py",
        "purge_pii.py",
        "purge_temp_files.py",
        "purge_tenant.py",
        "rotate_audit_logs.py",
    ]

    def test_shared_module_exists(self):
        mod_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "coordinator",
                "jobs",
                "litigation_hold.py",
            )
        )
        self.assertTrue(
            os.path.isfile(mod_path),
            f"Shared module not found at {mod_path}",
        )

    def test_all_commands_import_shared_check(self):
        """Every deletion command must import from jobs.litigation_hold."""
        for cmd_name in self._DELETION_COMMANDS:
            cmd_path = os.path.join(self._COMMANDS_DIR, cmd_name)
            with self.subTest(command=cmd_name):
                self.assertTrue(
                    os.path.isfile(cmd_path),
                    f"Command file not found: {cmd_path}",
                )
                with open(cmd_path, "r", encoding="utf-8") as f:
                    source = f.read()
                self.assertIn(
                    "from jobs.litigation_hold import",
                    source,
                    f"{cmd_name} does not import from the shared litigation_hold module",
                )

    def test_no_command_has_local_litigation_function(self):
        """No deletion command should define its own _is_litigation_hold_active."""
        for cmd_name in self._DELETION_COMMANDS:
            cmd_path = os.path.join(self._COMMANDS_DIR, cmd_name)
            with self.subTest(command=cmd_name):
                if not os.path.isfile(cmd_path):
                    continue
                with open(cmd_path, "r", encoding="utf-8") as f:
                    source = f.read()
                self.assertNotIn(
                    "def _is_litigation_hold_active",
                    source,
                    f"{cmd_name} still defines a local _is_litigation_hold_active function",
                )

    def test_tasks_file_imports_shared_module(self):
        """The Celery tasks file must import from the shared module."""
        tasks_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "coordinator",
                "jobs",
                "tasks.py",
            )
        )
        self.assertTrue(os.path.isfile(tasks_path))
        with open(tasks_path, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn(
            "from .litigation_hold import is_litigation_hold_active",
            source,
            "tasks.py does not import from the shared litigation_hold module",
        )

    def test_tasks_file_no_local_litigation_function(self):
        """The Celery tasks file must not define a local version."""
        tasks_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "coordinator",
                "jobs",
                "tasks.py",
            )
        )
        with open(tasks_path, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn(
            "def _is_litigation_hold_active",
            source,
            "tasks.py still defines a local _is_litigation_hold_active function",
        )


if __name__ == "__main__":
    unittest.main()
