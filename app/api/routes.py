"""FastAPI route handlers for the Auto Grading System API."""

from __future__ import annotations

import logging

import networkx as nx
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
    """Grade a student code submission using an LLM-driven pipeline.

    Pipeline:
    0. Gemini suy ra reference solution từ hướng làm của sinh viên + đề bài.
    1. Build CFG cho reference solution suy ra và code sinh viên.
    2. So sánh đồ thị (difference detection) → ComparisonResult.
    3. Phân loại lỗi + mức độ nghiêm trọng → List[ClassifiedError].
    4. Chạy sandbox để lấy lỗi runtime.
    5. Tạo repair guide từng bước.
    6. Gemini nhận TẤT CẢ 4 đầu vào để trả về điểm + feedback + summary.

    Args:
        request: :class:`~app.models.schemas.GradeRequest` payload.

    Returns:
        :class:`~app.models.schemas.GradeResponse` với điểm từ LLM, feedback,
        repair guide, execution result và graph diff.
    """
    gemini = GeminiClient()

    # Bước 0: Gemini suy ra reference solution từ hướng làm của sinh viên
    inferred_ref_code = await gemini.infer_reference_solution(
        request.question, request.student_code
    )

    # Bước 1: Build CFG cho cả hai
    try:
        ref_graph = _builder.build(inferred_ref_code)
    except SyntaxError:
        logger.warning("inferred_reference has syntax error; using empty graph.")
        ref_graph = nx.DiGraph()

    try:
        stu_graph = _builder.build(request.student_code)
    except SyntaxError:
        logger.warning("student_code has syntax error; using empty graph.")
        stu_graph = nx.DiGraph()

    # Bước 2: So sánh đồ thị (difference detection)
    comparison = _comparator.compare(ref_graph, stu_graph)

    # Bước 3: Phân loại lỗi + mức độ nghiêm trọng
    errors = _classifier.classify(comparison)

    # Bước 4: Chạy sandbox để lấy lỗi runtime
    execution_result = _executor.execute(request.student_code, request.test_cases)

    # Bước 5: Tạo repair guide từng bước
    repair_gen = RepairGuideGenerator(gemini)
    repair_guide = await repair_gen.generate(errors, inferred_ref_code, request.student_code)

    # Bước 6: LLM chấm điểm với TẤT CẢ 4 đầu vào
    grading_output = await gemini.grade_with_llm(
        question=request.question,
        student_code=request.student_code,
        inferred_reference=inferred_ref_code,
        comparison_result=comparison,
        errors=errors,
        repair_guide=repair_guide,
        execution_errors=execution_result.errors,
        runtime_error=execution_result.errors[0] if execution_result.errors else None,
    )

    return GradeResponse(
        score=grading_output.score,
        feedback=grading_output.feedback,
        repair_guide=repair_guide,
        execution_result=execution_result,
        graph_diff=comparison,
        errors=errors,
        inferred_reference=inferred_ref_code,
        summary=grading_output.summary,
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
