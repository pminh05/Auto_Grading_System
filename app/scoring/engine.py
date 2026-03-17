"""Scoring engine: compute a weighted score from graph diff and test results."""

from __future__ import annotations

from typing import Any, Dict, List

from app.models.schemas import (
    ClassifiedError,
    ComparisonResult,
    ErrorSeverity,
    ExecutionResult,
    ScoreResult,
)


class ScoringEngine:
    """Compute a 0–100 score from graph similarity and test execution results.

    Score formula
    -------------
    - Graph similarity component : 40 points  (similarity_score × 40)
    - Test-case pass-rate component : 40 points  (pass_rate × 40)
    - Code-quality component : 20 points  (20 - minor_error_deductions, ≥ 0)

    Deductions per classified error
    --------------------------------
    - CRITICAL : −15 points each
    - MAJOR    :  −8 points each
    - MINOR    :  −3 points each

    The final score is clamped to [0, 100].
    """

    # Weight constants
    GRAPH_WEIGHT: float = 40.0
    TEST_WEIGHT: float = 40.0
    QUALITY_WEIGHT: float = 20.0

    # Deduction constants
    DEDUCTION_CRITICAL: float = 15.0
    DEDUCTION_MAJOR: float = 8.0
    DEDUCTION_MINOR: float = 3.0

    def score(
        self,
        comparison_result: ComparisonResult,
        execution_result: ExecutionResult,
        errors: List[ClassifiedError] | None = None,
    ) -> ScoreResult:
        """Compute the final score for a student submission.

        Args:
            comparison_result: CFG comparison output containing
                ``similarity_score``.
            execution_result: Sandbox execution output containing
                ``passed`` and ``failed`` counts.
            errors: Optional list of classified errors used for deductions.
                If *None*, deductions are not applied.

        Returns:
            A fully-populated :class:`~app.models.schemas.ScoreResult`.
        """
        errors = errors or []

        # 1. Graph similarity component
        graph_score = comparison_result.similarity_score * self.GRAPH_WEIGHT

        # 2. Test-case pass rate component
        total_tests = execution_result.passed + execution_result.failed
        pass_rate = (execution_result.passed / total_tests) if total_tests > 0 else 0.0
        test_score = pass_rate * self.TEST_WEIGHT

        # 3. Code quality component (starts at max, reduced by MINOR errors)
        minor_count = sum(1 for e in errors if e.severity == ErrorSeverity.MINOR)
        quality_score = max(0.0, self.QUALITY_WEIGHT - minor_count * self.DEDUCTION_MINOR)

        # 4. Severity-based deductions applied to the first two components
        critical_count = sum(1 for e in errors if e.severity == ErrorSeverity.CRITICAL)
        major_count = sum(1 for e in errors if e.severity == ErrorSeverity.MAJOR)
        deductions = (
            critical_count * self.DEDUCTION_CRITICAL
            + major_count * self.DEDUCTION_MAJOR
        )

        raw_score = graph_score + test_score + quality_score - deductions
        total_score = max(0.0, min(100.0, raw_score))

        breakdown: Dict[str, Any] = {
            "graph_similarity": round(comparison_result.similarity_score, 4),
            "graph_score": round(graph_score, 2),
            "pass_rate": round(pass_rate, 4),
            "test_score": round(test_score, 2),
            "quality_score": round(quality_score, 2),
            "critical_errors": critical_count,
            "major_errors": major_count,
            "minor_errors": minor_count,
            "deductions": round(deductions, 2),
        }

        return ScoreResult(
            total_score=round(total_score, 2),
            graph_similarity_score=round(graph_score, 2),
            test_pass_score=round(test_score, 2),
            code_quality_score=round(quality_score, 2),
            deductions=round(deductions, 2),
            breakdown=breakdown,
        )
