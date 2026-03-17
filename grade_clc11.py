"""Batch grading script for CLC11 student submissions.

Loads CLC11topic.json (reference solutions) and CLC11cv.json (student
submissions) from the current working directory, runs the full Auto Grading
System pipeline for every submission, then writes CLC11_grades.json and
prints a summary to stdout.

Usage::

    python grade_clc11.py
"""

from __future__ import annotations

import ast
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

# Fallback test case used when we cannot derive a meaningful expected output.
DUMMY_TEST_CASE = TestCase(input="", expected_output="")


# ---------------------------------------------------------------------------
# Output-based test case helpers
# ---------------------------------------------------------------------------

# Module-level names that should never be auto-printed (they are library refs
# or internal helpers, not computed results).
_SKIP_PRINT_NAMES = frozenset({"np", "pd", "os", "sys", "math", "json", "re"})

# Auto-print wrapper injected after the student/reference code.
# It converts numpy arrays and pandas objects to plain Python lists for
# deterministic string comparison across different environments.
_PRINT_WRAPPER = """\

# --- auto-print block ---
def __fmt(v):
    try:
        import numpy as __np
        if isinstance(v, __np.ndarray):
            lst = v.tolist()
            def _rnd(x):
                if isinstance(x, float):
                    return round(x, 6)
                if isinstance(x, list):
                    return [_rnd(i) for i in x]
                return x
            return str(_rnd(lst))
    except Exception:
        pass
    try:
        import pandas as __pd
        if isinstance(v, __pd.DataFrame):
            return str(v.values.tolist())
        if isinstance(v, __pd.Series):
            return str(v.tolist())
    except Exception:
        pass
    if isinstance(v, float):
        return str(round(v, 6))
    return str(v)
"""


def _get_leaf_vars(stmts: list) -> List[str]:
    """Return top-level variable names that are "final outputs" in *stmts*.

    A variable is a leaf if:
    1. It is assigned at the top level (``Name`` target only).
    2. It is **not** used in the RHS of any statement that comes *after* its
       last assignment.
    3. It is not in ``_SKIP_PRINT_NAMES``.

    This ensures that only the true end-result variables are printed, not
    intermediate computations.  Using leaf variables for comparison makes the
    output-based test robust to different intermediate variable names used by
    the reference and student while still detecting wrong final values.

    Args:
        stmts: Module-level AST statement list (``ast.Module.body``).

    Returns:
        Leaf variable names ordered by the position of their last assignment.
    """
    # Track last assignment index for each Name target
    last_assigned_at: dict[str, int] = {}
    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id not in _SKIP_PRINT_NAMES:
                    last_assigned_at[target.id] = idx
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id not in _SKIP_PRINT_NAMES:
                last_assigned_at[stmt.target.id] = idx

    # Track every statement index where each Name is READ (used in any expression)
    used_in_stmt: dict[str, list[int]] = {}
    for idx, stmt in enumerate(stmts):
        # Walk ALL sub-expressions of the statement to find Name reads.
        # For Assign/AugAssign, walk the value (RHS); for other stmts walk everything.
        if isinstance(stmt, ast.Assign):
            expr_root = stmt.value
        elif isinstance(stmt, ast.AugAssign):
            expr_root = stmt.value
        else:
            expr_root = stmt
        for node in ast.walk(expr_root):
            if isinstance(node, ast.Name):
                used_in_stmt.setdefault(node.id, []).append(idx)

    # Leaf = last-assigned and never used in any later statement
    leaf_vars = []
    for var, last_idx in last_assigned_at.items():
        uses_after = [u for u in used_in_stmt.get(var, []) if u > last_idx]
        if not uses_after:
            leaf_vars.append(var)

    # Return in order of last assignment (most natural output order)
    return sorted(leaf_vars, key=lambda v: last_assigned_at[v])


def _wrap_code_for_output(code: str) -> str:
    """Append auto-print statements for leaf (final output) variables only.

    Parses *code* to identify module-level variables that are computed but
    not subsequently consumed (leaf variables in the data-dependency graph).
    Only those variables are printed, which makes the output comparison robust
    to different intermediate variable names between reference and student code.

    Values are emitted in a normalised format: numpy arrays converted to
    Python lists (floats rounded to 6 dp), pandas DataFrames/Series to lists.

    If *code* has a syntax error the original code is returned unchanged.

    Args:
        code: Python source code (reference or student solution).

    Returns:
        The original code with an appended auto-print block, or the original
        code unchanged on ``SyntaxError``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    leaf_vars = _get_leaf_vars(tree.body)

    if not leaf_vars:
        return code

    print_lines = [_PRINT_WRAPPER]
    for name in leaf_vars:
        print_lines.append(f"try: print(__fmt({name}))")
        print_lines.append(f"except (NameError, AttributeError, TypeError, ValueError): pass")

    return code + "\n".join(print_lines)


def _make_ref_test_case(
    ref_code: str,
    executor: SandboxExecutor,
) -> TestCase:
    """Run the reference solution (wrapped) and create a test case from its output.

    If the reference produces non-empty stdout the resulting
    :class:`~app.models.schemas.TestCase` uses that as ``expected_output``.
    An empty expected output means "just check the code runs without error".

    Args:
        ref_code: Reference solution source code.
        executor: A :class:`SandboxExecutor` for running the code.

    Returns:
        A :class:`~app.models.schemas.TestCase` with ``expected_output`` set
        to the reference stdout (normalised by :func:`_wrap_code_for_output`).
    """
    wrapped = _wrap_code_for_output(ref_code)
    probe = executor.execute(wrapped, [DUMMY_TEST_CASE])
    if probe.test_results:
        expected = probe.test_results[0].actual_output.rstrip("\n")
        return TestCase(
            input="",
            expected_output=expected,
            description="Reference output comparison",
        )
    return DUMMY_TEST_CASE


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

    # --- 1. PDG comparison ---------------------------------------------------
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
        # Derive the expected output by running the first valid reference
        # solution through the auto-print wrapper.  This gives output-based
        # testing even for problems that contain no explicit print() calls.
        test_case = DUMMY_TEST_CASE
        for ref_code in reference_solutions:
            tc = _make_ref_test_case(ref_code, executor)
            if tc.expected_output:  # non-empty → meaningful comparison
                test_case = tc
                break

        # Run student code through the same auto-print wrapper so that its
        # output can be compared with the reference's normalised output.
        wrapped_student = _wrap_code_for_output(student_code)
        exec_result = executor.execute(wrapped_student, [test_case])

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
