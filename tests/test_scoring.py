"""Unit tests for app/scoring/engine.py."""

from __future__ import annotations

import pytest

from app.models.schemas import (
    ClassifiedError,
    ComparisonResult,
    ErrorSeverity,
    ErrorType,
    ExecutionResult,
    TestCase,
    TestResult,
)
from app.scoring.engine import ScoringEngine


@pytest.fixture
def engine() -> ScoringEngine:
    return ScoringEngine()


def _make_comparison(similarity: float = 1.0) -> ComparisonResult:
    return ComparisonResult(similarity_score=similarity)


def _make_execution(passed: int = 0, failed: int = 0) -> ExecutionResult:
    return ExecutionResult(passed=passed, failed=failed)


def _error(severity: ErrorSeverity, error_type: ErrorType = ErrorType.ALGORITHM) -> ClassifiedError:
    return ClassifiedError(
        error_type=error_type,
        severity=severity,
        description="test error",
    )


class TestScoringEngine:
    """Tests for ScoringEngine.score()."""

    def test_perfect_score(self, engine: ScoringEngine) -> None:
        """Perfect graph similarity + all tests passing + no errors = 100."""
        comparison = _make_comparison(1.0)
        execution = _make_execution(passed=5, failed=0)
        result = engine.score(comparison, execution, errors=[])
        assert result.total_score == pytest.approx(100.0, abs=0.01)

    def test_zero_graph_similarity(self, engine: ScoringEngine) -> None:
        """Zero graph similarity contributes 0 to the graph component."""
        comparison = _make_comparison(0.0)
        execution = _make_execution(passed=5, failed=0)
        result = engine.score(comparison, execution, errors=[])
        assert result.graph_similarity_score == pytest.approx(0.0, abs=0.01)

    def test_no_tests_pass_rate_zero(self, engine: ScoringEngine) -> None:
        """With zero test cases the test pass rate contribution is 0."""
        comparison = _make_comparison(1.0)
        execution = _make_execution(passed=0, failed=0)
        result = engine.score(comparison, execution, errors=[])
        assert result.test_pass_score == pytest.approx(0.0, abs=0.01)

    def test_critical_error_deducts(self, engine: ScoringEngine) -> None:
        """One CRITICAL error should deduct 15 points."""
        comparison = _make_comparison(1.0)
        execution = _make_execution(passed=5, failed=0)
        errors = [_error(ErrorSeverity.CRITICAL)]
        result = engine.score(comparison, execution, errors=errors)
        assert result.deductions == pytest.approx(15.0, abs=0.01)

    def test_major_error_deducts(self, engine: ScoringEngine) -> None:
        """One MAJOR error should deduct 8 points."""
        comparison = _make_comparison(1.0)
        execution = _make_execution(passed=5, failed=0)
        errors = [_error(ErrorSeverity.MAJOR)]
        result = engine.score(comparison, execution, errors=errors)
        assert result.deductions == pytest.approx(8.0, abs=0.01)

    def test_minor_error_reduces_quality(self, engine: ScoringEngine) -> None:
        """One MINOR error should reduce code_quality_score by 3 points."""
        comparison = _make_comparison(1.0)
        execution = _make_execution(passed=5, failed=0)
        errors = [_error(ErrorSeverity.MINOR)]
        result = engine.score(comparison, execution, errors=errors)
        assert result.code_quality_score == pytest.approx(17.0, abs=0.01)

    def test_score_clamped_to_zero(self, engine: ScoringEngine) -> None:
        """Many critical errors should clamp the score to 0, not negative."""
        comparison = _make_comparison(0.0)
        execution = _make_execution(passed=0, failed=5)
        errors = [_error(ErrorSeverity.CRITICAL) for _ in range(20)]
        result = engine.score(comparison, execution, errors=errors)
        assert result.total_score >= 0.0

    def test_score_clamped_to_100(self, engine: ScoringEngine) -> None:
        """Score must never exceed 100."""
        comparison = _make_comparison(1.0)
        execution = _make_execution(passed=10, failed=0)
        result = engine.score(comparison, execution, errors=[])
        assert result.total_score <= 100.0

    def test_partial_test_pass_rate(self, engine: ScoringEngine) -> None:
        """50% test pass rate should contribute 20 points to test_pass_score."""
        comparison = _make_comparison(0.0)
        execution = _make_execution(passed=1, failed=1)
        result = engine.score(comparison, execution, errors=[])
        assert result.test_pass_score == pytest.approx(20.0, abs=0.01)

    def test_breakdown_contains_expected_keys(self, engine: ScoringEngine) -> None:
        """Score breakdown should contain all expected keys."""
        comparison = _make_comparison(0.8)
        execution = _make_execution(passed=3, failed=2)
        result = engine.score(comparison, execution, errors=[])
        for key in ("graph_similarity", "pass_rate", "graph_score", "test_score", "quality_score"):
            assert key in result.breakdown

    def test_no_errors_no_deductions(self, engine: ScoringEngine) -> None:
        """When errors list is empty deductions must be 0."""
        comparison = _make_comparison(0.5)
        execution = _make_execution(passed=2, failed=2)
        result = engine.score(comparison, execution, errors=[])
        assert result.deductions == pytest.approx(0.0, abs=0.01)
