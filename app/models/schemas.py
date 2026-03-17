"""Pydantic schemas (request/response models) for the Auto Grading System API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ErrorType(str, Enum):
    """Category of a detected error."""

    SYNTAX = "SYNTAX"
    LOOP = "LOOP"
    ALGORITHM = "ALGORITHM"
    DATA_HANDLING = "DATA_HANDLING"


class ErrorSeverity(str, Enum):
    """Severity level of a detected error."""

    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TestCase(BaseModel):
    """A single test case with input and expected output."""

    input: str = Field(..., description="Input data for the test case (stdin)")
    expected_output: str = Field(..., description="Expected stdout output")
    description: Optional[str] = Field(default=None, description="Human-readable description")


class TestResult(BaseModel):
    """Result of running a single test case."""

    test_case: TestCase
    actual_output: str = Field(default="", description="Actual stdout produced by student code")
    passed: bool = Field(default=False, description="Whether the test case passed")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    execution_time: float = Field(default=0.0, description="Wall-clock execution time in seconds")


class ClassifiedError(BaseModel):
    """A detected and classified error in student code."""

    error_type: ErrorType = Field(..., description="Category of the error")
    severity: ErrorSeverity = Field(..., description="Severity level")
    description: str = Field(..., description="Human-readable description of the error")
    line_number: Optional[int] = Field(default=None, description="Approximate line number in student code")
    node_label: Optional[str] = Field(default=None, description="Graph node label where error was detected")


class RepairStep(BaseModel):
    """A single step in the repair guide."""

    step_number: int = Field(..., description="Ordered step number (1-based)")
    description: str = Field(..., description="Description of what to fix")
    code_example: Optional[str] = Field(default=None, description="Example corrected code snippet")
    priority: int = Field(default=1, description="Priority (1 = highest)")


class RepairGuide(BaseModel):
    """Complete step-by-step repair guide for student code."""

    steps: List[RepairStep] = Field(default_factory=list, description="Ordered list of repair steps")
    summary: str = Field(default="", description="High-level summary of repairs needed")


class ComparisonResult(BaseModel):
    """Result of comparing reference and student CFGs."""

    missing_nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Nodes in reference but not in student")
    extra_nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Nodes in student but not in reference")
    missing_edges: List[Dict[str, Any]] = Field(default_factory=list, description="Edges in reference but not in student")
    extra_edges: List[Dict[str, Any]] = Field(default_factory=list, description="Edges in student but not in reference")
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Graph similarity score 0.0–1.0")


class ScoreResult(BaseModel):
    """Final score breakdown for a student submission."""

    total_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Final score out of 100")
    graph_similarity_score: float = Field(default=0.0, description="Score component from graph similarity (0–40)")
    test_pass_score: float = Field(default=0.0, description="Score component from test pass rate (0–40)")
    code_quality_score: float = Field(default=0.0, description="Score component from code quality (0–20)")
    deductions: float = Field(default=0.0, description="Total point deductions from errors")
    breakdown: Dict[str, Any] = Field(default_factory=dict, description="Detailed breakdown of score components")


class ExecutionResult(BaseModel):
    """Result of executing student code against all test cases."""

    test_results: List[TestResult] = Field(default_factory=list)
    passed: int = Field(default=0, description="Number of passed test cases")
    failed: int = Field(default=0, description="Number of failed test cases")
    errors: List[str] = Field(default_factory=list, description="Runtime errors encountered")
    total_execution_time: float = Field(default=0.0, description="Total execution time in seconds")


# ---------------------------------------------------------------------------
# API Request / Response models
# ---------------------------------------------------------------------------


class GradeRequest(BaseModel):
    """Request body for POST /api/v1/grade."""

    question: str = Field(..., description="The problem statement / question text")
    reference_solution: str = Field(..., description="Reference (model) solution source code")
    student_code: str = Field(..., description="Student's submitted source code")
    test_cases: List[TestCase] = Field(default_factory=list, description="Test cases to run against student code")


class GradeResponse(BaseModel):
    """Response body for POST /api/v1/grade."""

    score: ScoreResult
    feedback: str = Field(default="", description="LLM-generated natural language feedback")
    repair_guide: RepairGuide
    execution_result: ExecutionResult
    graph_diff: ComparisonResult
    errors: List[ClassifiedError] = Field(default_factory=list)


class ExecuteRequest(BaseModel):
    """Request body for POST /api/v1/execute."""

    code: str = Field(..., description="Python source code to execute")
    test_cases: List[TestCase] = Field(default_factory=list)


class ExecuteResponse(BaseModel):
    """Response body for POST /api/v1/execute."""

    results: List[TestResult] = Field(default_factory=list)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    """Request body for POST /api/v1/analyze."""

    reference_code: str = Field(..., description="Reference solution source code")
    student_code: str = Field(..., description="Student's source code")


class AnalyzeResponse(BaseModel):
    """Response body for POST /api/v1/analyze."""

    graph_diff: ComparisonResult
    errors: List[ClassifiedError] = Field(default_factory=list)
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
