"""FastAPI route handlers for the Auto Grading System API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.error.classifier import ErrorClassifier
from app.graph.builder import GraphBuilder
from app.graph.comparator import GraphComparator
from app.llm.gemini_client import GeminiClient
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ExecuteRequest,
    ExecuteResponse,
    GradeRequest,
    GradeResponse,
)
from app.repair.generator import RepairGuideGenerator
from app.sandbox.executor import SandboxExecutor
from app.scoring.engine import ScoringEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["grading"])

# Shared service instances (stateless, safe to reuse)
_builder = GraphBuilder()
_comparator = GraphComparator()
_classifier = ErrorClassifier()
_executor = SandboxExecutor()
_scorer = ScoringEngine()


@router.post("/grade", response_model=GradeResponse, summary="Grade a student submission")
async def grade(request: GradeRequest) -> GradeResponse:
    """Grade a student code submission against a reference solution.

    The pipeline:
    1. Build CFGs for reference and student code.
    2. Compare the graphs to detect structural differences.
    3. Classify the differences into typed, severity-rated errors.
    4. Execute the student code against the provided test cases.
    5. Compute a weighted score.
    6. Generate LLM feedback and a repair guide.

    Args:
        request: :class:`~app.models.schemas.GradeRequest` payload.

    Returns:
        :class:`~app.models.schemas.GradeResponse` with score, feedback,
        repair guide, execution result, and graph diff.

    Raises:
        :class:`fastapi.HTTPException`: On parse or internal errors.
    """
    try:
        # 1. Build graphs
        ref_graph = _builder.build(request.reference_solution)
        stu_graph = _builder.build(request.student_code)
    except SyntaxError as exc:
        raise HTTPException(status_code=422, detail=f"Syntax error in submitted code: {exc}") from exc

    # 2. Compare graphs
    comparison = _comparator.compare(ref_graph, stu_graph)

    # 3. Classify errors
    errors = _classifier.classify(comparison)

    # 4. Execute student code
    execution_result = _executor.execute(request.student_code, request.test_cases)

    # 5. Score
    score_result = _scorer.score(comparison, execution_result, errors)

    # 6. LLM feedback + repair guide
    gemini = GeminiClient()
    repair_gen = RepairGuideGenerator(gemini)

    feedback = await gemini.generate_feedback(request.question, request.student_code, errors)
    repair_guide = await repair_gen.generate(errors, request.reference_solution, request.student_code)

    return GradeResponse(
        score=score_result,
        feedback=feedback,
        repair_guide=repair_guide,
        execution_result=execution_result,
        graph_diff=comparison,
        errors=errors,
    )


@router.post("/execute", response_model=ExecuteResponse, summary="Execute code against test cases")
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    """Run student code in the sandbox against the provided test cases.

    Args:
        request: :class:`~app.models.schemas.ExecuteRequest` payload.

    Returns:
        :class:`~app.models.schemas.ExecuteResponse` with per-test results.
    """
    execution_result = _executor.execute(request.code, request.test_cases)
    return ExecuteResponse(
        results=execution_result.test_results,
        passed=execution_result.passed,
        failed=execution_result.failed,
        errors=execution_result.errors,
    )


@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyze structural differences between two submissions")
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Compare CFGs of reference and student code without full grading.

    Args:
        request: :class:`~app.models.schemas.AnalyzeRequest` payload.

    Returns:
        :class:`~app.models.schemas.AnalyzeResponse` with graph diff,
        classified errors, and similarity score.

    Raises:
        :class:`fastapi.HTTPException`: On syntax errors in either code block.
    """
    try:
        ref_graph = _builder.build(request.reference_code)
        stu_graph = _builder.build(request.student_code)
    except SyntaxError as exc:
        raise HTTPException(status_code=422, detail=f"Syntax error: {exc}") from exc

    comparison = _comparator.compare(ref_graph, stu_graph)
    errors = _classifier.classify(comparison)

    return AnalyzeResponse(
        graph_diff=comparison,
        errors=errors,
        similarity_score=comparison.similarity_score,
    )
