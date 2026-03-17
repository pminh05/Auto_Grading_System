"""Unit tests for app/sandbox/executor.py."""

from __future__ import annotations

import pytest

from app.models.schemas import TestCase
from app.sandbox.executor import SandboxExecutor


@pytest.fixture
def executor() -> SandboxExecutor:
    """Return a SandboxExecutor with a short timeout for tests."""
    return SandboxExecutor(timeout=5, max_memory_mb=64)


class TestSandboxExecutor:
    """Tests for SandboxExecutor.execute()."""

    def test_hello_world(self, executor: SandboxExecutor) -> None:
        """A simple Hello World program should pass a matching test case."""
        code = "print('Hello, World!')"
        test_cases = [TestCase(input="", expected_output="Hello, World!")]
        result = executor.execute(code, test_cases)
        assert result.passed == 1
        assert result.failed == 0

    def test_wrong_output_fails(self, executor: SandboxExecutor) -> None:
        """A program whose output does not match should fail the test case."""
        code = "print('wrong')"
        test_cases = [TestCase(input="", expected_output="correct")]
        result = executor.execute(code, test_cases)
        assert result.passed == 0
        assert result.failed == 1

    def test_stdin_input(self, executor: SandboxExecutor) -> None:
        """The executor should feed stdin to the student program."""
        code = "name = input()\nprint(f'Hello, {name}!')"
        test_cases = [TestCase(input="Alice", expected_output="Hello, Alice!")]
        result = executor.execute(code, test_cases)
        assert result.passed == 1

    def test_multiple_test_cases(self, executor: SandboxExecutor) -> None:
        """Multiple test cases should each be run independently."""
        code = "n = int(input())\nprint(n * 2)"
        test_cases = [
            TestCase(input="3", expected_output="6"),
            TestCase(input="5", expected_output="10"),
            TestCase(input="0", expected_output="0"),
        ]
        result = executor.execute(code, test_cases)
        assert result.passed == 3
        assert result.failed == 0

    def test_runtime_error_captured(self, executor: SandboxExecutor) -> None:
        """A program that raises an exception should be marked as failed."""
        code = "raise ValueError('oops')"
        test_cases = [TestCase(input="", expected_output="")]
        result = executor.execute(code, test_cases)
        assert result.failed >= 1
        assert len(result.errors) >= 1

    def test_timeout(self, executor: SandboxExecutor) -> None:
        """A program that runs longer than the timeout should be killed."""
        code = "import time\ntime.sleep(100)"
        test_cases = [TestCase(input="", expected_output="")]
        fast_executor = SandboxExecutor(timeout=1)
        result = fast_executor.execute(code, test_cases)
        assert result.failed == 1
        assert "timed out" in (result.errors[0] if result.errors else result.test_results[0].error or "").lower()

    def test_empty_test_cases(self, executor: SandboxExecutor) -> None:
        """Executing with no test cases should return a valid empty result."""
        code = "print('hi')"
        result = executor.execute(code, [])
        assert result.passed == 0
        assert result.failed == 0
        assert result.test_results == []

    def test_syntax_error_in_code(self, executor: SandboxExecutor) -> None:
        """Code with a syntax error should produce an error in the result."""
        code = "def broken(:\n    pass"
        test_cases = [TestCase(input="", expected_output="")]
        result = executor.execute(code, test_cases)
        assert result.failed >= 1

    def test_execution_time_recorded(self, executor: SandboxExecutor) -> None:
        """Execution time should be non-negative for each test result."""
        code = "print('x')"
        test_cases = [TestCase(input="", expected_output="x")]
        result = executor.execute(code, test_cases)
        for tr in result.test_results:
            assert tr.execution_time >= 0.0
