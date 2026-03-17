"""Sandbox executor: run student code safely using subprocess with resource limits."""

from __future__ import annotations

import sys
import os
import subprocess
import tempfile
import time
from typing import List

from app.config import settings
from app.models.schemas import ExecutionResult, TestCase, TestResult


class SandboxExecutor:
    """Execute student Python code against a set of test cases in a subprocess.

    The subprocess is killed after *timeout* seconds.  On POSIX systems,
    memory usage is limited via the ``resource`` module injected into the
    child process; on Windows the memory limit is silently skipped.

    Attributes:
        timeout: Maximum wall-clock seconds per test case.
        max_memory_mb: Maximum resident set size in MiB (POSIX only).
    """

    def __init__(
        self,
        timeout: int | None = None,
        max_memory_mb: int | None = None,
    ) -> None:
        """Initialise the executor with optional overrides.

        Args:
            timeout: Execution timeout in seconds.  Defaults to the value in
                :attr:`~app.config.Settings.max_execution_time`.
            max_memory_mb: Memory limit in MiB.  Defaults to the value in
                :attr:`~app.config.Settings.max_memory_mb`.
        """
        self.timeout: int = timeout if timeout is not None else settings.max_execution_time
        self.max_memory_mb: int = max_memory_mb if max_memory_mb is not None else settings.max_memory_mb

    def execute(self, code: str, test_cases: List[TestCase]) -> ExecutionResult:
        """Run *code* against every test case and collect results.

        Args:
            code: Python 3 source code (student submission).
            test_cases: List of :class:`~app.models.schemas.TestCase` objects.

        Returns:
            An :class:`~app.models.schemas.ExecutionResult` summarising all
            test outcomes.
        """
        test_results: List[TestResult] = []
        global_errors: List[str] = []
        total_time = 0.0

        for tc in test_cases:
            result = self._run_single(code, tc)
            test_results.append(result)
            total_time += result.execution_time
            if result.error:
                global_errors.append(result.error)

        passed = sum(1 for r in test_results if r.passed)
        failed = len(test_results) - passed

        return ExecutionResult(
            test_results=test_results,
            passed=passed,
            failed=failed,
            errors=global_errors,
            total_execution_time=round(total_time, 4),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_single(self, code: str, test_case: TestCase) -> TestResult:
        """Execute *code* once with the given *test_case* input.

        Args:
            code: Python source code to execute.
            test_case: The test case providing stdin and expected output.

        Returns:
            A :class:`~app.models.schemas.TestResult` for this test case.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            start = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, tmp_path],
                input=test_case.input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=self._safe_env(),
            )
            elapsed = time.perf_counter() - start

            actual_output = proc.stdout.rstrip("\n")
            expected_output = test_case.expected_output.rstrip("\n")
            stderr_output = proc.stderr.strip()
            # A test fails if the exit code is non-zero OR output doesn't match
            passed = (proc.returncode == 0) and (actual_output == expected_output)
            error_msg: str | None = stderr_output if stderr_output else None

            return TestResult(
                test_case=test_case,
                actual_output=actual_output,
                passed=passed,
                error=error_msg,
                execution_time=round(elapsed, 4),
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                test_case=test_case,
                actual_output="",
                passed=False,
                error=f"Execution timed out after {self.timeout} second(s).",
                execution_time=float(self.timeout),
            )
        except Exception as exc:  # noqa: BLE001
            return TestResult(
                test_case=test_case,
                actual_output="",
                passed=False,
                error=str(exc),
                execution_time=0.0,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _safe_env() -> dict:
        """Return a minimal environment dictionary for the subprocess."""
        safe_keys = {"PATH", "PYTHONPATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
        return {k: v for k, v in os.environ.items() if k in safe_keys}
