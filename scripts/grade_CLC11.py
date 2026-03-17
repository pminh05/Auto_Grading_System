#!/usr/bin/env python3
"""
Batch grading script cho lớp CLC11.

Usage:
    python scripts/grade_CLC11.py [--output-dir OUTPUT_DIR] [--email EMAIL] [--slug SLUG]

Output:
    output/CLC11_results.csv   - bảng điểm tổng hợp
    output/CLC11_results.json  - kết quả chi tiết với feedback
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_loader import DataLoader, SubmissionEntry, TopicEntry
from app.graph.builder import GraphBuilder
from app.graph.comparator import GraphComparator
from app.error.classifier import ErrorClassifier
from app.models.schemas import (
    ComparisonResult,
    ErrorSeverity,
    ExecutionResult,
)
from app.scoring.engine import ScoringEngine


# ---------------------------------------------------------------------------
# Scoring helpers (no sandbox / no LLM for batch speed)
# ---------------------------------------------------------------------------

def _grade_submission(
    topic: TopicEntry,
    submission: SubmissionEntry,
    builder: GraphBuilder,
    comparator: GraphComparator,
    classifier: ErrorClassifier,
    engine: ScoringEngine,
) -> dict:
    """Grade a single submission and return a result dict."""
    email = submission.email
    slug = submission.slug
    has_error = submission.has_runtime_error

    # Default values
    similarity = 0.0
    comparison: ComparisonResult = ComparisonResult()
    errors = []
    build_failed = False

    # Build CFGs only if no runtime error was reported
    if not has_error:
        ref_code = topic.solutions[0]
        try:
            ref_graph = builder.build(ref_code)
            stu_graph = builder.build(submission.solution)
            comparison = comparator.compare(ref_graph, stu_graph)
            similarity = comparison.similarity_score
            errors = classifier.classify(comparison)
        except SyntaxError:
            build_failed = True
            similarity = 0.0
            comparison = ComparisonResult()
            errors = []
    else:
        # Runtime error: try to build CFG for graph score anyway
        ref_code = topic.solutions[0]
        try:
            ref_graph = builder.build(ref_code)
            stu_graph = builder.build(submission.solution)
            comparison = comparator.compare(ref_graph, stu_graph)
            similarity = comparison.similarity_score
            errors = classifier.classify(comparison)
        except SyntaxError:
            build_failed = True
            similarity = 0.0
            comparison = ComparisonResult()
            errors = []

    # Execution score: 0 if runtime error, otherwise 40
    execution_score = 0.0 if has_error else 40.0

    # Graph score from similarity
    graph_score = similarity * engine.GRAPH_WEIGHT

    # Quality score
    minor_count = sum(1 for e in errors if e.severity == ErrorSeverity.MINOR)
    quality_score = max(0.0, engine.QUALITY_WEIGHT - minor_count * engine.DEDUCTION_MINOR)

    # Severity deductions
    critical_count = sum(1 for e in errors if e.severity == ErrorSeverity.CRITICAL)
    major_count = sum(1 for e in errors if e.severity == ErrorSeverity.MAJOR)

    # If build_failed and has_error → add a CRITICAL error for syntax
    if build_failed:
        critical_count += 1

    deductions = critical_count * engine.DEDUCTION_CRITICAL + major_count * engine.DEDUCTION_MAJOR
    raw_total = graph_score + execution_score + quality_score - deductions
    total_score = max(0.0, min(100.0, raw_total))

    errors_summary = "; ".join(
        f"[{e.severity.value}] {e.description[:60]}" for e in errors[:5]
    )
    if build_failed:
        errors_summary = "[CRITICAL] SyntaxError: could not parse student code" + (
            ("; " + errors_summary) if errors_summary else ""
        )

    return {
        "email": email,
        "slug": slug,
        "has_runtime_error": has_error,
        "graph_score": round(graph_score, 2),
        "execution_score": round(execution_score, 2),
        "quality_score": round(quality_score, 2),
        "total_score": round(total_score, 2),
        "similarity": round(similarity, 4),
        "error_count_critical": critical_count,
        "error_count_major": major_count,
        "error_count_minor": minor_count,
        "errors_summary": errors_summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch grading script for CLC11")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output"),
        help="Directory to write output files (default: output/)",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Only grade submissions for this email",
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="Only grade submissions for this slug",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    loader = DataLoader()
    topics = loader.topics()
    submissions = loader.submissions()

    # Apply filters
    if args.email:
        submissions = [s for s in submissions if s.email == args.email]
    if args.slug:
        submissions = [s for s in submissions if s.slug == args.slug]

    if not submissions:
        print("No submissions found for the given filters.")
        sys.exit(0)

    # Pipeline components
    builder = GraphBuilder()
    comparator = GraphComparator()
    classifier = ErrorClassifier()
    engine = ScoringEngine()

    results: list[dict] = []
    total = len(submissions)
    print(f"Grading {total} submissions...")

    for i, submission in enumerate(submissions, 1):
        topic = topics.get(submission.slug)
        if topic is None:
            print(f"  [{i}/{total}] SKIP {submission.email} / {submission.slug} — topic not found")
            continue

        result = _grade_submission(topic, submission, builder, comparator, classifier, engine)
        results.append(result)

        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] processed...")

    print(f"\nGraded {len(results)} submissions.")

    # Write CSV
    csv_path = output_dir / "CLC11_results.csv"
    csv_columns = [
        "email", "slug", "has_runtime_error",
        "graph_score", "execution_score", "quality_score", "total_score",
        "similarity", "error_count_critical", "error_count_major",
        "error_count_minor", "errors_summary",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV written: {csv_path}")

    # Write JSON
    json_path = output_dir / "CLC11_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON written: {json_path}")

    # Print summary
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    """Print a human-readable summary to stdout."""
    if not results:
        return

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total submissions graded: {len(results)}")

    # Average score per slug
    from collections import defaultdict
    slug_scores: dict[str, list[float]] = defaultdict(list)
    email_scores: dict[str, list[float]] = defaultdict(list)
    slug_errors: dict[str, int] = defaultdict(int)

    for r in results:
        slug_scores[r["slug"]].append(r["total_score"])
        email_scores[r["email"]].append(r["total_score"])
        if r["has_runtime_error"]:
            slug_errors[r["slug"]] += 1

    print("\nĐiểm trung bình theo bài:")
    for slug, scores in slug_scores.items():
        short = slug.split("-")[0] + "-" + slug.split("-")[1]
        print(f"  {short}: {sum(scores)/len(scores):.1f} (n={len(scores)})")

    print("\nĐiểm trung bình theo sinh viên (top 5):")
    avg_by_email = {
        email: sum(scores) / len(scores)
        for email, scores in email_scores.items()
    }
    top5 = sorted(avg_by_email.items(), key=lambda x: x[1], reverse=True)[:5]
    for email, avg in top5:
        print(f"  {email}: {avg:.1f}")

    print("\nTỉ lệ lỗi runtime cao nhất theo bài:")
    slugs_with_scores = [s for s in slug_scores if slug_scores[s]]
    if slugs_with_scores:
        highest_slug = max(
            slugs_with_scores,
            key=lambda s: slug_errors[s] / len(slug_scores[s]),
        )
        rate = slug_errors[highest_slug] / len(slug_scores[highest_slug]) * 100
        short = highest_slug.split("-")[0] + "-" + highest_slug.split("-")[1]
        print(f"  {short}: {rate:.1f}% lỗi runtime")
    print("=" * 60)


if __name__ == "__main__":
    main()
