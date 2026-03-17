"""Batch grading script for CLC11 student submissions.

Loads CLC11topic.json (reference solutions) and CLC11cv.json (student
submissions) from the current working directory, runs the full Auto Grading
System pipeline for every submission, then writes CLC11_grades.json and
prints a summary to stdout.

Usage::

    python grade_clc11.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

from app.error.classifier import ErrorClassifier
from app.graph.builder import GraphBuilder
from app.graph.comparator import GraphComparator
from app.models.schemas import (
    ClassifiedError,
    ComparisonResult,
    ExecutionResult,
    TestCase,
    TestResult,
)
from app.sandbox.executor import SandboxExecutor
from app.scoring.engine import ScoringEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOPIC_FILE = "CLC11topic.json"
CV_FILE = "CLC11cv.json"
OUTPUT_FILE = "CLC11_grades.json"

# A single dummy test case used to probe whether the student code runs cleanly.
# We do NOT match output — a submission is considered "passed" when the
# subprocess exits with code 0 (no runtime/syntax errors).
DUMMY_TEST_CASE = TestCase(input="", expected_output="")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_comparison() -> ComparisonResult:
    """Return a zero-similarity ComparisonResult for un-parseable submissions."""
    return ComparisonResult(
        missing_nodes=[],
        extra_nodes=[],
        missing_edges=[],
        extra_edges=[],
        similarity_score=0.0,
    )


def _failed_execution() -> ExecutionResult:
    """Return an ExecutionResult that reflects a complete execution failure."""
    dummy_result = TestResult(
        test_case=DUMMY_TEST_CASE,
        actual_output="",
        passed=False,
        error="Exited with error status 1",
        execution_time=0.0,
    )
    return ExecutionResult(
        test_results=[dummy_result],
        passed=0,
        failed=1,
        errors=["Exited with error status 1"],
        total_execution_time=0.0,
    )


def _best_comparison(
    student_code: str,
    reference_solutions: List[str],
    builder: GraphBuilder,
    comparator: GraphComparator,
) -> ComparisonResult:
    """Build student CFG and compare against every reference; return best match.

    Args:
        student_code: The student's Python source code.
        reference_solutions: List of reference solution strings for the problem.
        builder: A :class:`GraphBuilder` instance (reused for efficiency).
        comparator: A :class:`GraphComparator` instance.

    Returns:
        The :class:`ComparisonResult` with the highest similarity score, or a
        zero-similarity result if the student code cannot be parsed.
    """
    try:
        student_graph = builder.build(student_code)
    except SyntaxError:
        return _empty_comparison()

    best: Optional[ComparisonResult] = None
    for ref_code in reference_solutions:
        try:
            ref_graph = builder.build(ref_code)
        except SyntaxError:
            continue
        result = comparator.compare(ref_graph, student_graph)
        if best is None or result.similarity_score > best.similarity_score:
            best = result

    return best if best is not None else _empty_comparison()


def _grade_submission(
    submission: Dict[str, Any],
    topics: Dict[str, Any],
    builder: GraphBuilder,
    comparator: GraphComparator,
    classifier: ErrorClassifier,
    executor: SandboxExecutor,
    engine: ScoringEngine,
) -> Dict[str, Any]:
    """Grade a single student submission and return a result dict.

    Args:
        submission: A student submission object from CLC11cv.json.
        topics: The loaded CLC11topic.json mapping slug → topic data.
        builder: Shared :class:`GraphBuilder`.
        comparator: Shared :class:`GraphComparator`.
        classifier: Shared :class:`ErrorClassifier`.
        executor: Shared :class:`SandboxExecutor`.
        engine: Shared :class:`ScoringEngine`.

    Returns:
        A dictionary with grading results ready to be serialised to JSON.
    """
    email: str = submission["email"]
    slug: str = submission["slug"]
    student_code: str = submission.get("solution", "")
    submission_error: Optional[str] = submission.get("error")

    topic = topics.get(slug, {})
    reference_solutions: List[str] = topic.get("solutions", [])

    # --- 1. CFG comparison ---------------------------------------------------
    comparison_result = _best_comparison(
        student_code, reference_solutions, builder, comparator
    )

    # --- 2. Error classification ---------------------------------------------
    errors: List[ClassifiedError] = classifier.classify(comparison_result)

    # --- 3. Sandbox execution ------------------------------------------------
    if submission_error is not None:
        # The submission is already known to fail at runtime; skip execution.
        exec_result = _failed_execution()
    else:
        exec_result = executor.execute(student_code, [DUMMY_TEST_CASE])
        # Re-evaluate "passed" purely on exit-code 0 (no stderr/timeout errors).
        # The dummy expected_output is "" so the default logic may mark a
        # clean-but-printing run as failed — we correct that here.
        output_mismatch_only = len(exec_result.errors) == 0 and exec_result.failed == 1
        if output_mismatch_only and exec_result.test_results:
            tr = exec_result.test_results[0]
            if tr.error is None:
                # Exit code was 0, only "failure" was output mismatch → pass.
                exec_result = ExecutionResult(
                    test_results=[
                        TestResult(
                            test_case=tr.test_case,
                            actual_output=tr.actual_output,
                            passed=True,
                            error=None,
                            execution_time=tr.execution_time,
                        )
                    ],
                    passed=1,
                    failed=0,
                    errors=[],
                    total_execution_time=exec_result.total_execution_time,
                )

    # --- 4. Scoring ----------------------------------------------------------
    score_result = engine.score(comparison_result, exec_result, errors)

    # --- 5. Serialise errors -------------------------------------------------
    serialised_errors = [
        {
            "type": e.error_type.value,
            "severity": e.severity.value,
            "description": e.description,
        }
        for e in errors
    ]

    return {
        "email": email,
        "slug": slug,
        "score": score_result.total_score,
        "graph_similarity": comparison_result.similarity_score,
        "tests_passed": exec_result.passed,
        "tests_total": exec_result.passed + exec_result.failed,
        "errors": serialised_errors,
        "execution_error": submission_error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: load data, grade submissions, write output."""
    # --- Load input files ----------------------------------------------------
    try:
        with open(TOPIC_FILE, encoding="utf-8") as fh:
            topics: Dict[str, Any] = json.load(fh)
    except FileNotFoundError:
        sys.exit(f"ERROR: Cannot find {TOPIC_FILE} in the current directory.")

    try:
        with open(CV_FILE, encoding="utf-8") as fh:
            submissions: List[Dict[str, Any]] = json.load(fh)
    except FileNotFoundError:
        sys.exit(f"ERROR: Cannot find {CV_FILE} in the current directory.")

    print(f"Loaded {len(topics)} problem(s) and {len(submissions)} submission(s).")

    # --- Initialise pipeline components -------------------------------------
    builder = GraphBuilder()
    comparator = GraphComparator()
    classifier = ErrorClassifier()
    executor = SandboxExecutor()
    engine = ScoringEngine()

    # --- Grade each submission -----------------------------------------------
    grades: List[Dict[str, Any]] = []

    for idx, submission in enumerate(submissions, start=1):
        email = submission.get("email", "unknown")
        slug = submission.get("slug", "unknown")
        print(f"[{idx}/{len(submissions)}] Grading {email} / {slug} ...", flush=True)

        try:
            result = _grade_submission(
                submission, topics, builder, comparator, classifier, executor, engine
            )
        except (SyntaxError, ValueError, TypeError, OSError) as exc:
            print(f"  ERROR grading submission: {exc}", file=sys.stderr)
            result = {
                "email": email,
                "slug": slug,
                "score": 0.0,
                "graph_similarity": 0.0,
                "tests_passed": 0,
                "tests_total": 1,
                "errors": [
                    {
                        "type": "ALGORITHM",
                        "severity": "CRITICAL",
                        "description": f"Grading pipeline error: {exc}",
                    }
                ],
                "execution_error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 — catch-all to keep batch running
            print(f"  UNEXPECTED ERROR grading submission: {exc}", file=sys.stderr)
            result = {
                "email": email,
                "slug": slug,
                "score": 0.0,
                "graph_similarity": 0.0,
                "tests_passed": 0,
                "tests_total": 1,
                "errors": [
                    {
                        "type": "ALGORITHM",
                        "severity": "CRITICAL",
                        "description": f"Grading pipeline error: {exc}",
                    }
                ],
                "execution_error": str(exc),
            }

        grades.append(result)
        print(
            f"  → score={result['score']:.1f}, "
            f"similarity={result['graph_similarity']:.2f}, "
            f"tests={result['tests_passed']}/{result['tests_total']}"
        )

    # --- Write output file ---------------------------------------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(grades, fh, ensure_ascii=False, indent=2)
    print(f"\nGrades written to {OUTPUT_FILE}")

    # --- Print summary statistics --------------------------------------------
    total = len(grades)
    if total == 0:
        print("No submissions processed.")
        return

    avg_score = sum(g["score"] for g in grades) / total
    error_count = sum(1 for g in grades if g["execution_error"] is not None)
    clean_count = total - error_count

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total submissions processed : {total}")
    print(f"Average score               : {avg_score:.2f}")
    print(f"Clean runs (no exec error)  : {clean_count}")
    print(f"Execution errors            : {error_count}")

    # Per-problem averages
    problem_scores: Dict[str, List[float]] = defaultdict(list)
    for g in grades:
        problem_scores[g["slug"]].append(g["score"])

    print("\nPer-problem average scores:")
    for slug, scores in sorted(problem_scores.items()):
        prob_avg = sum(scores) / len(scores)
        print(f"  {slug}: {prob_avg:.2f} (n={len(scores)})")

    print("=" * 60)


if __name__ == "__main__":
    main()
